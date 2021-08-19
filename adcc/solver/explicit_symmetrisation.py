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
from adcc.AmplitudeVector import AmplitudeVector, QED_AmplitudeVector, gs_vec
import numpy as np

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
                #print(b)
                vec[b] = evaluate(self.symmetrisation_functions[b](vec[b]))
            return vec

        if isinstance(new_vectors, AmplitudeVector):
            return self.symmetrise([new_vectors])[0]
        elif isinstance(new_vectors[0], QED_AmplitudeVector):
            # we dont have to symmetrise the gs blocks...actually only the pphh blocks are symmetrised here
            if "pphh" in new_vectors[0].elec.blocks_ph:
                test_list = new_vectors
                for ind, vec in enumerate(new_vectors):
                    #if vec.elec.pphh != evaluate(self.symmetrisation_functions["pphh"](vec.elec.pphh)): #vec.elec.pphh != symm_subroutine(vec.elec).pphh:
                    #    print("something changed")
                    #else:
                    #    print("nothing changed")
                    #vec = QED_AmplitudeVector(gs=vec.gs, ph=vec.elec.ph, pphh=evaluate(self.symmetrisation_functions["pphh"](vec.elec.pphh)), 
                    #                        gs1=vec.gs1, ph1=vec.phot.ph, pphh1=evaluate(self.symmetrisation_functions["pphh"](vec.phot.pphh)),
                    #                        gs2=vec.gs2, ph2=vec.phot2.ph, pphh2=evaluate(self.symmetrisation_functions["pphh"](vec.phot2.pphh)))
                    test_list[ind].elec = symm_subroutine(vec.elec) #evaluate(self.symmetrisation_functions["pphh"](vec.elec.pphh))
                    test_list[ind].phot = symm_subroutine(vec.phot) #evaluate(self.symmetrisation_functions["pphh"](vec.phot.pphh))
                    test_list[ind].phot2 = symm_subroutine(vec.phot2) #evaluate(self.symmetrisation_functions["pphh"](vec.phot2.pphh))
                if test_list[0].elec.pphh == new_vectors[0].elec.pphh:
                    print("in symm nothing changed")
                    if new_vectors[0].elec.pphh != evaluate(self.symmetrisation_functions["pphh"](test_list[0].elec.pphh)):
                        print("but something changed for the symmetrization, which was not passed to the QED_AmplitudeVector")
                else:
                    print("in symm something changed")
                #if vec.elec.pphh == evaluate(self.symmetrisation_functions["pphh"](vec.elec.pphh)):
                #    print("symm still yields no change")
                #print(type(evaluate(self.symmetrisation_functions["pphh"](vec.elec.pphh))), evaluate(self.symmetrisation_functions["pphh"](vec.elec.pphh)).shape)
                #if new_vec.elec == symm_subroutine(vec.elec):
                #    print("correctly changed in symmetrise function")
                #elif new_vec.elec == vec.elec:
                #    print("nothing changed in symmetrise function")
                #else:
                #    print("not correctly changed in symmetrise function")
                #    diff = new_vec.elec - symm_subroutine(vec.elec)
                #    print("squared norm of difference with symmetrise = ", np.sqrt(diff @ diff))
                #    diff2 = new_vec.elec - vec.elec
                #    print("squared norm of difference without symmetrise = ", np.sqrt(diff2 @ diff2))
                #    diff3 = vec.elec - symm_subroutine(vec.elec)
                #    print("squared norm of difference between no symmetrise and symmetrise = ", np.sqrt(diff3 @ diff3)) # why is this zero??????????
                #    #print(type(new_vec.elec.pphh), type(symm_subroutine(vec.elec).pphh))
                #vec.elec = symm_subroutine(vec.elec)
                #vec.phot = symm_subroutine(vec.phot)
                #vec.phot2 = symm_subroutine(vec.phot2)
        elif isinstance(new_vectors[0], AmplitudeVector):
            for vec in new_vectors:
                vec = symm_subroutine(vec)
            #if not isinstance(vec, AmplitudeVector):
            #    raise TypeError("new_vectors has to be an "
            #                    "iterable of AmplitudeVector")
            #for b in vec.blocks_ph:
            #    if b not in self.symmetrisation_functions:
            #        continue
            #    vec[b] = evaluate(self.symmetrisation_functions[b](vec[b]))
        return new_vectors


class IndexSpinSymmetrisation(IndexSymmetrisation):
    """
    Enforce both the required index symmetry as well as an additional
    explicit spin symmetry in the new amplitude vectors.
    """
    def __init__(self, matrix, enforce_spin_kind="singlet"):
        super().__init__(matrix)
        self.enforce_spin_kind = enforce_spin_kind
        print("index spin symm class is used")

    def symmetrise(self, new_vectors):
        if isinstance(new_vectors, AmplitudeVector):
            return self.symmetrise([new_vectors])[0]
        new_vectors = super().symmetrise(new_vectors)

        # Enforce singlet (or other spin_kind) spin in the doubles block
        # of all amplitude vectors
        for vec in new_vectors:
            # Only work on the doubles part
            # the other blocks are not yet implemented
            # or nothing needs to be done ("ph" block)
            if "pphh" in vec.blocks_ph:
                # TODO: Note that the "d" is needed here because the C++ side
                #       does not yet understand ph and pphh
                amplitude_vector_enforce_spin_kind(
                    vec.pphh, "d", self.enforce_spin_kind
                )
        return new_vectors


IndexSpinSymmetrisation.symmetrise.__doc__ = \
    IndexSymmetrisation.symmetrise.__doc__
