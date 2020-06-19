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

from numpy import sqrt
from .guess import guess_zero
from itertools import groupby

from libadcc import MoIndexTranslation


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
    n_searched_for = max(3, n_elements)
    while True:
        res = []
        n_searched_for *= 2

        found = tensor.select_n_min(n_searched_for)
        for index, value in found:
            telem = TensorElement(motrans, index, value)
            if predicate(telem):
                res.append(telem)

        if len(found) < n_searched_for:
            # We will not be able to find more because already
            # less found than requested
            break

        if len(res) >= n_elements:
            break  # Everything found

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
    motrans = MoIndexTranslation(matrix.mospaces, matrix.block_spaces("s"))
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
        return (ret[0]["s"].is_allowed(telem.index)
                and telem.spin_change == spin_change)

    elements = find_smallest_matching_elements(
        pred_singles, matrix.diagonal("s"), motrans, n_guesses,
        degeneracy_tolerance=degeneracy_tolerance
    )
    if len(elements) == 0:
        return []

    # By construction of find_smallest_elements the returned elements
    # are already sorted such that adjacent vectors of equal value
    # only differ in spin indices if possible and further are sorted
    # by element value, so we can just group by the value.
    ivec = 0
    for key, group in groupby(elements, key=lambda telem: telem.value):
        group = list(group)
        if len(group) == 1:  # Just add the single vector
            ret[ivec]["s"][group[0].index] = 1.0
            ivec += 1
        elif len(group) == 2:
            # Since these two are grouped together, their
            # spatial parts must be identical.

            # Add the positive linear combination ...
            ret[ivec]["s"][group[0].index] = 1 / sqrt(2)
            ret[ivec]["s"][group[1].index] = 1 / sqrt(2)
            ivec += 1

            # ... and the negative linear combination
            if ivec < n_guesses:
                ret[ivec]["s"][group[0].index] = 1 / sqrt(2)
                ret[ivec]["s"][group[1].index] = -1 / sqrt(2)
                ivec += 1
        else:
            raise AssertionError("group size > 3 should not occur "
                                 "when setting up single guesse.")
    assert ivec == n_guesses

    # Resize in case less guesses found than requested
    return ret[:ivec]
