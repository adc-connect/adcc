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
from adcc.AmplitudeVector import AmplitudeVector


class PreconditionerIdentity:
    """
    Preconditioner, which does absolutely nothing
    """
    def apply(self, invecs):
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

        self.diagonal = adcmatrix.diagonal()
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

    def apply(self, invecs):
        if isinstance(invecs, AmplitudeVector):
            if not isinstance(self.shifts, (float, np.number)):
                raise TypeError("Can only apply JacobiPreconditioner "
                                "to a single vector if shifts is "
                                "only a single number.")
            return invecs / (self.diagonal - self.shifts)
        elif isinstance(invecs, list):
            if len(self.shifts) != len(invecs):
                raise ValueError("Number of vectors passed does not agree "
                                 "with number of shifts stored inside "
                                 "precoditioner. Update using the "
                                 "'update_shifts' method.")

            return [v / (self.diagonal - self.shifts[i])
                    for i, v in enumerate(invecs)]
        else:
            raise TypeError("Input type not understood: " + str(type(invecs)))

    def __matmul__(self, invecs):
        return self.apply(invecs)

    # __matvec__
