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
from libadcc import amplitude_vector_enforce_spin_kind

from adcc import evaluate
from adcc.AmplitudeVector import AmplitudeVector, QED_AmplitudeVector

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
        self.symmetrisation_functions = \
            matrix.construct_symmetrisation_for_blocks()
        print("index symm class init is used")

    def symmetrise(self, new_vectors):
        """
        Symmetrise a set of new vectors to be added to the subspace.

        new_vectors          Vectors to symmetrise (updated in-place)

        Returns:
            The updated new_vectors
        """

        def symm_subroutine(vec):
            if not isinstance(vec, AmplitudeVector):
                    raise TypeError("new_vectors has to be an "
                                    "iterable of AmplitudeVector")
            for b in vec.blocks_ph:
                if b not in self.symmetrisation_functions:
                    continue
                vec[b] = evaluate(self.symmetrisation_functions[b](vec[b]))
            return vec

        if isinstance(new_vectors, (AmplitudeVector, QED_AmplitudeVector)):
            return self.symmetrise([new_vectors])[0]
        elif isinstance(new_vectors[0], QED_AmplitudeVector):
            if "pphh" in new_vectors[0].elec.blocks_ph:
                for vec in new_vectors:
                    vec.elec = self.symmetrise([vec.elec])[0]
                    vec.phot = self.symmetrise([vec.phot])[0]
                    vec.phot2 = self.symmetrise([vec.phot2])[0]
        elif isinstance(new_vectors[0], AmplitudeVector):
            for vec in new_vectors:
                vec = symm_subroutine(vec)
            if not isinstance(vec, AmplitudeVector):
                raise TypeError("new_vectors has to be an "
                                "iterable of AmplitudeVector")
            for b in vec.blocks_ph:
                if b not in self.symmetrisation_functions:
                    continue
                vec[b] = evaluate(self.symmetrisation_functions[b](vec[b]))
        return new_vectors


class IndexSpinSymmetrisation(IndexSymmetrisation):
    """
    Enforce both the required index symmetry as well as an additional
    explicit spin symmetry in the new amplitude vectors.
    """
    def __init__(self, matrix, enforce_spin_kind="singlet"):
        super().__init__(matrix)
        self.enforce_spin_kind = enforce_spin_kind

    def symmetrise(self, new_vectors):
        if isinstance(new_vectors, (AmplitudeVector, QED_AmplitudeVector)):
            return self.symmetrise([new_vectors])[0]

        new_vectors = super().symmetrise(new_vectors)

        # Enforce singlet (or other spin_kind) spin in the doubles block
        # of all amplitude vectors
        for vec in new_vectors:
            if isinstance(vec, QED_AmplitudeVector):
                if "pphh" in vec.elec.blocks_ph:
                    amplitude_vector_enforce_spin_kind(
                        vec.elec.pphh, "d", self.enforce_spin_kind
                    )
                    amplitude_vector_enforce_spin_kind(
                        vec.phot.pphh, "d", self.enforce_spin_kind
                    )
                    amplitude_vector_enforce_spin_kind(
                        vec.phot2.pphh, "d", self.enforce_spin_kind
                    )
            # Only work on the doubles part
            # the other blocks are not yet implemented
            # or nothing needs to be done ("ph" block)
            elif "pphh" in vec.blocks_ph:
                # TODO: Note that the "d" is needed here because the C++ side
                #       does not yet understand ph and pphh
                amplitude_vector_enforce_spin_kind(
                    vec.pphh, "d", self.enforce_spin_kind
                )
        return new_vectors


IndexSpinSymmetrisation.symmetrise.__doc__ = \
    IndexSymmetrisation.symmetrise.__doc__
