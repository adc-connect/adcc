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
import libadcc
import numpy as np

from adcc import evaluate
from libadcc import MoIndexTranslation
from itertools import groupby

from ..AdcMatrix import AdcMatrixlike
from .guess_zero import guess_zero


def guesses_from_diagonal(matrix, n_guesses, block="ph", spin_change=0,
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
                 (typically "ph" or "pphh").
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

    if block not in matrix.axis_blocks:
        raise ValueError("The passed ADC matrix does not have the block '{}.'"
                         "".format(block))
    if n_guesses == 0:
        return []

    if block == "ph":
        guessfunction = guesses_from_diagonal_singles
    elif block == "pphh":
        guessfunction = guesses_from_diagonal_doubles
    else:
        raise ValueError(f"Don't know how to generate guesses for block {block}")

    return guessfunction(matrix, n_guesses, spin_change,
                         spin_block_symmetrisation, degeneracy_tolerance)


class TensorElement:
    def __init__(self, motrans, index, value):
        """
        Initialise a TensorElement from an MoIndexTranslation object,
        a tensor index and the corresponding value.
        """
        # TODO One could probably rewrite motrans to only take the data from
        #      the axis info the tensors have anyway. This way one could
        #      initialise such a TensorElement just from a tensor and an
        #      index, which seems like a lot more reasonable interface.
        self.index = tuple(index)
        self.subspaces = motrans.subspaces
        self.value = value

        splitted = motrans.split_spin(tuple(index))
        self.spin_block, self.block_index_spatial, self.inblock_index = splitted

    @property
    def spin_change(self):
        """
        Compute the change in spin induced by an excitation vector element
        referenced by the given tensor element. Assumes that an occupied
        index implies that an electron is taken out of such an object and that
        a virtual index implies that an electron is put into such an object in
        an excitation.
        """
        mapping_spin_change = {
            ("o", "a"): -0.5,  # remove alpha
            ("o", "b"): +0.5,  # remove beta
            ("v", "a"): +0.5,  # add    alpha
            ("v", "b"): -0.5,  # add    beta
        }
        return int(sum(mapping_spin_change[(space[0], spin)]
                       for space, spin in zip(self.subspaces, self.spin_block)))

    def __repr__(self):
        return f"({self.index}  {self.value})"


def find_smallest_matching_elements(predicate, tensor, motrans, n_elements,
                                    degeneracy_tolerance=1e-12):
    """
    Search for the n smallest elements in the passed tensor adhering to the
    passed predicate. If the returned vector contains less elements than
    requested, no more elements matching the criterion can be found. If it
    contains more elements, than the last set of elements beyond the requested
    number have the same value.
    """
    # Search for the n_elements smallest elements in the tensor
    # skipping over elements, which are not fulfilling the passed criteria.
    n_searched_for = max(10, 2 * n_elements + 6)
    while True:
        res = []

        found = tensor.select_n_min(n_searched_for)
        for index, value in found:
            telem = TensorElement(motrans, index, value)
            if predicate(telem):
                res.append(telem)

        if len(res) >= n_elements:
            break  # Everything found

        if len(found) < n_searched_for:
            # We will not be able to find more because already
            # less found than requested
            break
        else:
            n_searched_for *= 2  # Increase for the next round

    if len(res) == 0:
        return []

    # Sort elements in res ascendingly and by spatial indices, i.e. such
    # that blocks differing only in spin will be sorted adjacent
    def telem_nospin(telem):
        return (telem.value, telem.block_index_spatial, telem.inblock_index)

    res = sorted(res, key=telem_nospin)

    # Normalise the tensor values
    istart = 0
    for i in range(len(res)):
        if abs(res[istart].value - res[i].value) > degeneracy_tolerance:
            # Set to the average value
            avg = np.average([r.value for r in res[istart:i]])
            for j in range(istart, i):
                res[j].value = avg
            istart = i
    avg = np.average([r.value for r in res[istart:]])
    for j in range(istart, len(res)):
        res[j].value = avg

    if len(res) > n_elements:
        # Delete the extra elements, excluding the ones of identical value
        return [
            telem for telem in res if telem.value <= res[n_elements - 1].value
        ]
    else:
        return res


def guesses_from_diagonal_singles(matrix, n_guesses, spin_change=0,
                                  spin_block_symmetrisation="none",
                                  degeneracy_tolerance=1e-14):
    motrans = MoIndexTranslation(matrix.mospaces, matrix.axis_spaces["ph"])
    if n_guesses == 0:
        return []

    # Create a result vector of zero vectors with appropriate symmetry setup
    ret = [guess_zero(matrix, spin_change=spin_change,
                      spin_block_symmetrisation=spin_block_symmetrisation)
           for _ in range(n_guesses)]

    # Search of the smallest elements
    # This predicate checks an index is an allowed element for the singles
    # part of the guess vectors and has the requested spin-change
    def pred_singles(telem):
        return (ret[0].ph.is_allowed(telem.index)
                and telem.spin_change == spin_change)

    elements = find_smallest_matching_elements(
        pred_singles, matrix.diagonal().ph, motrans, n_guesses,
        degeneracy_tolerance=degeneracy_tolerance
    )
    if len(elements) == 0:
        return []

    # By construction of find_smallest_elements the returned elements
    # are already sorted such that adjacent vectors of equal value
    # only differ in spin indices (or they consider excitations from
    # degenerate orbitals). We want to form spin-adapted linear combinations
    # for the case of an unrestricted reference in the following
    # and therefore to exclude the spatial degeneracies we sort explicitly
    # only by value, and spatial indices (and not by spin).

    def telem_nospin(telem):
        return (telem.value, telem.block_index_spatial, telem.inblock_index)

    ivec = 0
    for value, group in groupby(elements, key=telem_nospin):
        if ivec >= len(ret):
            break

        group = list(group)
        if len(group) == 1:  # Just add the single vector
            ret[ivec].ph[group[0].index] = 1.0
            ivec += 1
        elif len(group) == 2:
            # Since these two are grouped together, their
            # spatial parts must be identical.

            # Add the positive linear combination ...
            ret[ivec].ph[group[0].index] = 1
            ret[ivec].ph[group[1].index] = 1
            ivec += 1

            # ... and the negative linear combination
            if ivec < n_guesses:
                ret[ivec].ph[group[0].index] = +1
                ret[ivec].ph[group[1].index] = -1
                ivec += 1
        else:
            raise AssertionError("group size > 3 should not occur "
                                 "when setting up single guesse.")
    assert ivec <= n_guesses

    # Resize in case less guesses found than requested
    # and normalise vectors
    return [evaluate(v / np.sqrt(v @ v)) for v in ret[:ivec]]


def guesses_from_diagonal_doubles(matrix, n_guesses, spin_change=0,
                                  spin_block_symmetrisation="none",
                                  degeneracy_tolerance=1e-14):
    if n_guesses == 0:
        return []

    # Create a result vector of zero vectors with appropriate symmetry setup
    ret = [guess_zero(matrix, spin_change=spin_change,
                      spin_block_symmetrisation=spin_block_symmetrisation)
           for _ in range(n_guesses)]

    # Build delta-Fock matrices
    spaces_d = matrix.axis_spaces["pphh"]
    df02 = matrix.ground_state.df(spaces_d[0] + spaces_d[2])
    df13 = matrix.ground_state.df(spaces_d[1] + spaces_d[3])

    guesses_d = [gv.pphh for gv in ret]  # Extract doubles parts
    spin_change_twice = int(spin_change * 2)
    assert spin_change_twice / 2 == spin_change
    n_found = libadcc.fill_pp_doubles_guesses(
        guesses_d, matrix.mospaces, df02, df13,
        spin_change_twice, degeneracy_tolerance
    )

    # Resize in case less guesses found than requested
    return ret[:n_found]

# TODO Generic algorithm for forming orthogonal spin components of arbitrary
#      size. Could be useful for the doubles guesses later on.
# if ivec + len(group) <= n_guesses:
#     # Just add all of them with element 1.0
#     for gvec in group:
#         ret[ivec].ph[gvec.index] = 1.0
#         ivec += 1
# else:
#     # Generate orthogonal coefficients for the linear combination
#     # of guess vectors: Make sure every guess vector has a
#     # non-zero component
#     coeff = np.eye(len(group))
#     coeff[:, 0] = 1
#     coeff, _ = np.linalg.qr(coeff)  # orthogonalise

#     for ifrom, ito in enumerate(range(ivec, n_guesses)):
#         for j in range(len(group)):
#             ret[ito].ph[group[j].index] = -coeff[j, ifrom]
#         ivec += 1
