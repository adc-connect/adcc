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
from adcc import empty_like
from adcc.AmplitudeVector import AmplitudeVector

from libadcc import amplitude_vector_enforce_spin_kind

# TODO
#    This interface is not that great and leads to duplicate information
#    (e.g. once for setting up the guesses and once for setting up the
#     explicit symmetrisation)
#    Maybe one could pass the guesses to these classes or construct
#    these classes along with the guesses or allow to do the guess symmetry
#    setup first and then use this symmetry setup for setting up both the
#    guesses and these classes (which is probably the best case)


class IndexSymmetrisation():
    """
    Enforce the very index symmetrisation required for a particular
    ADC matrix at hand in the new amplitude vectors.
    """
    def __init__(self, matrix):
        # Build symmetrisation functions required to be executed
        # for the respective block
        self.sym_for_block = matrix.construct_symmetrisation_for_blocks()

    def symmetrise(self, new_vectors, existing_subspace):
        """
        Symmetrise a set of new vectors to be added to the subspace.

        new_vectors          Vectors to symmetrise (updated in-place)
        existing_subspace    Existing subspace to take as a template

        Returns:
            The updated new_vectors

        It should be noted:
          - No orthogonalisation is performed
          - Only new_vectors are operated upon, existing_subspace
            is assumed to be properly symmetrised already.
          - Does not add new_vectors to the existing_subspace object
        """
        if not isinstance(existing_subspace[0], AmplitudeVector):
            raise TypeError("existing_subspace has to be an "
                            "iterable of AmplitudeVector")

        for vec in new_vectors:
            if not isinstance(vec, AmplitudeVector):
                raise TypeError("new_vectors has to be an "
                                "iterable of AmplitudeVector")
            for b in vec.blocks:
                if b not in self.sym_for_block:
                    continue
                index_symm_func = self.sym_for_block[b]

                # Create an empty block from the subspace vectors
                # since this has the appropriate symmetry set up
                symmetrised = empty_like(existing_subspace[0][b])
                #
                # symmetrise what we have (syntax is function(in, out)
                index_symm_func(vec[b], symmetrised)
                #
                # Substitute inside the passed amplitude vector
                vec[b] = symmetrised
        return new_vectors


class IndexSpinSymmetrisation(IndexSymmetrisation):
    """
    Enforce both the required index symmetry as well as an additional
    explicit spin symmetry in the new amplitude vectors.
    """
    def __init__(self, matrix, enforce_spin_kind="singlet"):
        super().__init__(matrix)
        self.enforce_spin_kind = enforce_spin_kind

    def symmetrise(self, new_vectors, existing_subspace):
        new_vectors = super().symmetrise(new_vectors, existing_subspace)

        # Enforce singlet (or other spin_kind) spin in the doubles block
        # of all amplitude vectors
        for vec in new_vectors:
            # Only work on the doubles part
            # the other blocks are not yet implemented
            # or nothing needs to be done ("s" block)
            [amplitude_vector_enforce_spin_kind(vec[b], b,
                                                self.enforce_spin_kind)
             for b in vec.blocks if b in ["d"]
             ]
        return new_vectors


IndexSpinSymmetrisation.symmetrise.__doc__ = \
    IndexSymmetrisation.symmetrise.__doc__
