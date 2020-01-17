#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
from adcc import AdcMatrixlike, AmplitudeVector, Tensor

import libadcc

__all__ = ["guess_zero", "guesses_from_diagonal",
           "guesses_singlet", "guesses_triplet", "guesses_any",
           "guesses_spin_flip"]


def guess_zero(matrix, irrep="A", spin_change=0,
               spin_block_symmetrisation="none"):
    """
    Return an AmplitudeVector object filled with zeros, but where the symmetry
    has been properly set up to meet the specified requirements on the guesses.

    matrix       The matrix for which guesses are to be constructed
    irrep        String describing the irreducable representation to consider.
    spin_change  The spin change to enforce in an excitation.
                 Typical values are 0 (singlet/triplet/any) and -1 (spin-flip).
    spin_block_symmetrisation
                 Symmetrisation to enforce between equivalent spin blocks, which
                 all yield the desired spin_change. E.g. if spin_change == 0,
                 then both the alpha->alpha and beta->beta blocks of the singles
                 part of the excitation vector achieve a spin change of 0.
                 The symmetry specified with this parameter will then be imposed
                 between the a-a and b-b blocks. Valid values are "none",
                 "symmetric" and "antisymmetric", where "none" enforces
                 no particular symmetry.
    """
    return AmplitudeVector(*tuple(
        Tensor(sym) for sym in guess_symmetries(
            matrix, irrep=irrep, spin_change=spin_change,
            spin_block_symmetrisation=spin_block_symmetrisation
        )
    ))


def guess_symmetries(matrix, irrep="A", spin_change=0,
                     spin_block_symmetrisation="none"):
    """
    Return guess symmetry objects (one for each AmplitudeVector block) such
    that the specified requirements on the guesses are satisfied.

    matrix       The matrix for which guesses are to be constructed
    irrep        String describing the irreducable representation to consider.
    spin_change  The spin change to enforce in an excitation.
                 Typical values are 0 (singlet/triplet/any) and -1 (spin-flip).
    spin_block_symmetrisation
                 Symmetrisation to enforce between equivalent spin blocks, which
                 all yield the desired spin_change. E.g. if spin_change == 0,
                 then both the alpha->alpha and beta->beta blocks of the singles
                 part of the excitation vector achieve a spin change of 0.
                 The symmetry specified with this parameter will then be imposed
                 between the a-a and b-b blocks. Valid values are "none",
                 "symmetric" and "antisymmetric", where "none" enforces
                 no particular symmetry.
    """
    if not isinstance(matrix, AdcMatrixlike):
        raise TypeError("matrix needs to be of type AdcMatrixlike")
    if spin_block_symmetrisation not in ["none", "symmetric", "antisymmetric"]:
        raise ValueError("Invalid value for spin_block_symmetrisation: "
                         "{}".format(spin_block_symmetrisation))
    if spin_block_symmetrisation != "none" and \
       not matrix.reference_state.restricted:
        raise ValueError("spin_block_symmetrisation != none is only valid for "
                         "ADC calculations on top of restricted reference "
                         "states.")
    if int(spin_change * 2) / 2 != spin_change:
        raise ValueError("Only integer or half-integer spin_change is allowed. "
                         "You passed {}".format(spin_change))
    if irrep != "A":
        raise NotImplementedError("Currently only irrep == 'A' is implemented.")

    gkind = libadcc.AdcGuessKind(irrep, float(spin_change),
                                 spin_block_symmetrisation)
    symmetries = libadcc.guess_symmetries(matrix.to_cpp(), gkind)

    # FIXME There are cases (e.g. when a AdcBlockView is employed), where
    #       there returned guess vectors contain a block, which is actually
    #       removed by the view. This corrects for that. When the guess selection
    #       has fully migrated to the python world, this should be removed.
    if len(symmetries) == 2 and "d" not in matrix.blocks:
        symmetries = [symmetries[0]]
    return symmetries


def guesses_from_diagonal(matrix, n_guesses, block="s",
                          irrep="A", spin_change=0,
                          spin_block_symmetrisation="none",
                          degeneracy_tolerance=1e-14):
    """
    Obtain guesses by inspecting a block of the diagonal of the passed ADC
    matrix. The symmetry of the returned vectors is already set-up properly.
    Note that this routine may return fewer vectors than requested in case the
    requested number could not be found.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "s" or "d").
    irrep        String describing the irreducable representation to consider.
    spin_change  The spin change to enforce in an excitation.
                 Typical values are 0 (singlet/triplet/any) and -1 (spin-flip).
    spin_block_symmetrisation
                 Symmetrisation to enforce between equivalent spin blocks, which
                 all yield the desired spin_change. E.g. if spin_change == 0,
                 then both the alpha->alpha and beta->beta blocks of the singles
                 part of the excitation vector achieve a spin change of 0.
                 The symmetry specified with this parameter will then be imposed
                 between the a-a and b-b blocks. Valid values are "none",
                 "symmetric" and "antisymmetric", where "none" enforces
                 no particular symmetry.
    degeneracy_tolerance
                 Tolerance for two entries of the diagonal to be considered
                 degenerate, i.e. identical.
    """
    if not isinstance(matrix, AdcMatrixlike):
        raise TypeError("matrix needs to be of type AdcMatrixlike")
    if spin_block_symmetrisation not in ["none", "symmetric", "antisymmetric"]:
        raise ValueError("Invalid value for spin_block_symmetrisation: "
                         "{}".format(spin_block_symmetrisation))
    if spin_block_symmetrisation != "none" and \
       not matrix.reference_state.restricted:
        raise ValueError("spin_block_symmetrisation != none is only valid for "
                         "ADC calculations on top of restricted reference "
                         "states.")
    if int(spin_change * 2) / 2 != spin_change:
        raise ValueError("Only integer or half-integer spin_change is allowed. "
                         "You passed {}".format(spin_change))

    if not matrix.has_block(block):
        raise ValueError("The passed ADC matrix does not have the block '{}.'"
                         "".format(block))
    if irrep != "A":
        raise NotImplementedError("Currently only irrep == 'A' is implemented.")

    if n_guesses == 0:
        return []

    gkind = libadcc.AdcGuessKind(irrep, float(spin_change),
                                 spin_block_symmetrisation)
    guesses = libadcc.guesses_from_diagonal(matrix.to_cpp(), gkind, block,
                                            n_guesses, degeneracy_tolerance)

    # FIXME There are cases (e.g. when a AdcBlockView is employed), where
    #       there returned guess vectors contain a block, which is actually
    #       removed by the view. This corrects for that. When the guess selection
    #       has fully migrated to the python world, this should be removed.
    if len(guesses[0].to_tuple()) == 2 and "d" not in matrix.blocks:
        return [AmplitudeVector(gv.to_tuple()[0]) for gv in guesses]
    else:
        return [AmplitudeVector(*gv.to_tuple()) for gv in guesses]


#
# Specialisations of guesses_from_diagonal for common cases
#
def guesses_singlet(matrix, n_guesses, block="s", irrep="A", **kwargs):
    """
    Obtain guesses for computing singlet states by inspecting the passed
    ADC matrix.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "s" or "d").
    irrep        String describing the irreducable representation to consider.
    kwargs       Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(matrix, n_guesses, block=block, irrep=irrep,
                                 spin_block_symmetrisation="symmetric",
                                 spin_change=0, **kwargs)


def guesses_triplet(matrix, n_guesses, block="s", irrep="A", **kwargs):
    """
    Obtain guesses for computing triplet states by inspecting the passed
    ADC matrix.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "s" or "d").
    irrep        String describing the irreducable representation to consider.
    kwargs       Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(matrix, n_guesses, block=block, irrep=irrep,
                                 spin_block_symmetrisation="antisymmetric",
                                 spin_change=0, **kwargs)


# guesses for computing any state (singlet or triplet)
guesses_any = guesses_from_diagonal


def guesses_spin_flip(matrix, n_guesses, block="s", irrep="A", **kwargs):
    """
    Obtain guesses for computing spin-flip states by inspecting the passed
    ADC matrix.

    matrix       The matrix for which guesses are to be constructed
    n_guesses    The number of guesses to be searched for. Less number of
                 vectors are returned if this many could not be found.
    block        Diagonal block to use for obtaining the guesses
                 (typically "s" or "d").
    irrep        String describing the irreducable representation to consider.
    kwargs       Any other argument understood by guesses_from_diagonal.
    """
    return guesses_from_diagonal(matrix, n_guesses, block=block, irrep=irrep,
                                 spin_change=-1, **kwargs)
