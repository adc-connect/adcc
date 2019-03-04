#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

from adcc import empty_like, AmplitudeVector
from libadcc import amplitude_vector_enforce_spin_kind


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

        new_vectors          Vectors to symmetrise
        existing_subspace    Existing subspace to take as a template

        It should be noted:
          - No orthogonalisation is performed
          - Only new_vectors are operated upon, existing_subspace
            is assumed to be properly symmetrised already.
          - Does not add new_vectors to the existing_subspace object
        """
        for vec in new_vectors:
            if not isinstance(vec, AmplitudeVector):
                raise TypeError("new_vectors has to be an "
                                "iterable of AmplitudeVector")

            for b, index_symm_func in self.sym_for_block.items():
                # Create an empty block from the subspace vectors
                # since this has the appropriate symmetry set up
                symmetrised = empty_like(existing_subspace[0][b])
                #
                # symmetrise what we have (syntax is function(in, out)
                index_symm_func(vec[b], symmetrised)
                #
                # Substitute inside the passed amplitude vector
                vec[b] = symmetrised


class IndexSpinSymmetrisation(IndexSymmetrisation):
    """
    Enforce both the required index symmetry as well as an additional
    explicit spin symmetry in the new amplitude vectors.
    """
    def __init__(self, matrix, enforce_spin_kind="singlet"):
        super().__init__(matrix)
        self.enforce_spin_kind = enforce_spin_kind

    def symmetrise(self, new_vectors, existing_subspace):
        super().symmetrise(new_vectors, existing_subspace)

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


IndexSpinSymmetrisation.symmetrise.__doc__ = \
    IndexSymmetrisation.symmetrise.__doc__
