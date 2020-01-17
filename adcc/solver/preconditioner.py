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
import numpy as np

from adcc.AdcMatrix import AdcMatrixlike
from adcc.functions import divide, empty_like, ones_like
from adcc.AmplitudeVector import AmplitudeVector


class PreconditionerIdentity:
    """
    Preconditioner, which does absolutely nothing
    """
    def apply(self, invecs, outvecs=None):
        """
        Apply preconditioner to a bunch of input vectors
        """
        if outvecs is not None:
            invecs.copy_to(outvecs)
        return invecs

    def __matmul__(self, x):
        return x


class JacobiPreconditioner:
    """
    Jacobi-type preconditioner

    Represents the application of (D - σ I)^{-1}, where
    D is the diagonal of the adcmatrix.
    """
    def __init__(self, adcmatrix, shifts=0.0):
        if not isinstance(adcmatrix, AdcMatrixlike):
            raise TypeError("Only an AdcMatrixlike may be used with this "
                            "preconditioner for now.")

        self.diagonal = AmplitudeVector(*tuple(
            adcmatrix.diagonal(block) for block in adcmatrix.blocks
        ))
        self.shifts = shifts

    def update_shifts(self, shifts):
        """
        Update the shift value or values applied to the diagonal.
        If this is a single value it will be applied to all
        vectors simultaneously. If it is multiple values,
        then each value will be applied only to one
        of the passed vectors.
        """
        self.shifts = shifts
        # TODO: this seems to be implemented?
        # if isinstance(shifts, (float, np.number)):
        #     raise NotImplementedError("Using only a single common shift is "
        #                               "not implemented at the moment.")

    def __compute_single_matvec(self, shift, invec, outvec):
        eps = 1e-6  # Epsilon factor to make sure that 1 / (shift - diagonal)
        #             does not become ill-conditioned as soon as the shift
        #             approaches the actual diagonal values (which are the
        #             eigenvalues for the ADC(2) doubles part if the coupling
        #             block are absent)
        shifted_diagonal = (self.diagonal
                            - (shift - eps) * ones_like(self.diagonal))
        divide(invec, shifted_diagonal, outvec)
        return outvec

    def apply(self, invecs, outvecs=None):
        if isinstance(invecs, AmplitudeVector):
            if outvecs is None:
                outvecs = empty_like(invecs)

            if not isinstance(self.shifts, (float, np.number)):
                raise TypeError("Can only apply JacobiPreconditioner "
                                "to a single vector if shifts is "
                                "only a single number.")
            return self.__compute_single_matvec(self.shifts, invecs, outvecs)
        elif isinstance(invecs, list):
            if outvecs is None:
                outvecs = [empty_like(v) for v in invecs]

            if len(self.shifts) != len(invecs):
                raise ValueError("Number of vectors passed does not agree "
                                 "with number of shifts stored inside "
                                 "precoditioner. Update using the "
                                 "'update_shifts' method.")

            for i in range(len(invecs)):
                self.__compute_single_matvec(self.shifts[i],
                                             invecs[i], outvecs[i])
            return outvecs
        else:
            raise TypeError("Input type not understood: " + str(type(invecs)))

    def __matmul__(self, invecs):
        return self.apply(invecs)

    # __matvec__
