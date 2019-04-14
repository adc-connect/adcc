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
from .AdcMethod import AdcMethod
from .functions import empty_like
from .AmplitudeVector import AmplitudeVector

import libadcc


class AdcMatrix(libadcc.AdcMatrix):
    def __init__(self, method, mp_results):
        """
        Initialise an ADC matrix from a method, the reference_state
        and appropriate MP results.
        """
        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)
        self.method = method
        super().__init__(method.name, mp_results)

    def compute_matvec(self, in_ampl, out_ampl=None):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result
        in the out_ampl if it is given, else the result
        will be returned.
        """
        if out_ampl is None:
            out_ampl = empty_like(in_ampl)
        elif not isinstance(out_ampl, type(in_ampl)):
            raise TypeError("Types of in_ample and out_ampl do not match.")
        if not isinstance(in_ampl, AmplitudeVector):
            raise TypeError("in_ampl has to be of type AmplitudeVector.")
        else:
            super().compute_matvec(in_ampl.to_cpp(), out_ampl.to_cpp())
        return out_ampl

    @property
    def ndim(self):
        return 2

    def matvec(self, v):
        out = empty_like(v)
        self.compute_matvec(v, out)
        return out

    def rmatvec(self, v):
        # ADC matrix is symmetric
        return self.matvec(self, v)

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.compute_matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.compute_matvec(ov) for ov in other]
        return NotImplemented

    def __repr__(self):
        return "AdcMatrix(method={})".format(self.method.name)

    def construct_symmetrisation_for_blocks(self):
        """
        Construct the symmetrisation functions, which need to be
        applied to relevant blocks of an AmplitudeVector in order
        to symmetrise it to the right symmetry in order to be used
        with the various matrix-vector-products of this function.

        Most importantly the returned functions antisymmetrise
        the occupied and virtual parts of the doubles parts
        if this is sensible for the method behind this adcmatrix.

        Returns a dictionary block identifier -> function
        """
        ret = {}
        if self.method.is_core_valence_separated:
            def symmetrise_cvs_adc_doubles(invec, outvec):
                # CVS doubles part is antisymmetric wrt. (i,K,a,b) <-> (i,K,b,a)
                invec.antisymmetrise_to(outvec, [(2, 3)])
            ret["d"] = symmetrise_cvs_adc_doubles
        else:
            def symmetrise_generic_adc_doubles(invec, outvec):
                scratch = empty_like(outvec)
                # doubles part is antisymmetric wrt. (i,j,a,b) <-> (i,j,b,a)
                invec.antisymmetrise_to(scratch, [(2, 3)])
                # doubles part is symmetric wrt. (i,j,a,b) <-> (j,i,b,a)
                scratch.symmetrise_to(outvec, [(0, 1), (2, 3)])
            ret["d"] = symmetrise_generic_adc_doubles
        return ret
